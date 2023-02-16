#include <ubinos.h>

#if (UBINOS__BSP__BOARD_MODEL == UBINOS__BSP__BOARD_MODEL__NUCLEOF207ZG)

#include "main.h"
#include "stm32f2xx_it.h"

/**
 * @brief  This function handles DTTY_STM32_UART interrupt request.
 * @param  None
 * @retval None
 */
void DTTY_STM32_UART_IRQHandler(void)
{
    HAL_UART_IRQHandler(&DTTY_STM32_UART_HANDLE);
}

#endif /* (UBINOS__BSP__BOARD_MODEL == UBINOS__BSP__BOARD_MODEL__NUCLEOF207ZG) */

